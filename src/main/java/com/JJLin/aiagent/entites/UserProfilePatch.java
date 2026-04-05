package com.JJLin.aiagent.entites;

import lombok.Data;

@Data
public class UserProfilePatch {

    private Integer age;
    private String occupation;
    private String education;
    private String incomeRange;
    private String gender;
    private String city;
    private String maritalStatus;
    private String notes;

    public boolean isEmpty() {
        return age == null
                && isBlank(occupation)
                && isBlank(education)
                && isBlank(incomeRange)
                && isBlank(gender)
                && isBlank(city)
                && isBlank(maritalStatus)
                && isBlank(notes);
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}